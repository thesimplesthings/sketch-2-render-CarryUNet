# sketch-2-render-CarryUNet

## Use Case
In SaaS game production, items are constantly created within the same art style.
For example, DarkOrbit spaceships.
Designing concepts for these variations is a repetitive and recurring task.

![image](https://github.com/user-attachments/assets/4780cc6c-841e-4d16-a788-2934a96affd4)

## The Tool
A lightweight CNN enables artists to train and generate new artwork from sketches.
They maintain control over shapes and structures, while detail rendering is accelerated—supporting faster and more efficient design exploration.

![image](https://github.com/user-attachments/assets/d1fe85fa-39c2-4b20-b11a-69620ccd2ab9)

## The Challenge
Extract details from few information.

![image](https://github.com/user-attachments/assets/3c8628e6-15a3-4f03-8369-df8437e105e5)

Carry U-Net uses the first decoding layer to transfer abstract, dense information to later layers, instead of relying on the classic U-Net skip connections.

![image](https://github.com/user-attachments/assets/793043bf-ccb1-48fa-9dc8-3a8b54cae274)

### Preventing MAE collapse on light weight CNN

![image](https://github.com/user-attachments/assets/ff5e444f-fc8d-45dc-8449-7766ca8ac897)

![image](https://github.com/user-attachments/assets/f5cbaec1-666b-4aca-924d-7801db9e025a)


## Results on 256 resolution & 12Million Params

![image](https://github.com/user-attachments/assets/0e8299ad-51d0-4180-bd18-9037f287261a)
![image](https://github.com/user-attachments/assets/be36b7e8-1b0f-494f-97d6-f7bb14d3b34a)
![image](https://github.com/user-attachments/assets/574235be-c6f0-413e-8933-99986f07baa4)
![image](https://github.com/user-attachments/assets/4665bfd3-a2d4-42b3-8770-8dfad9ab3eba)
